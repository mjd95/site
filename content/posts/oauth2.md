---
title: "OAuth2 By Example"
date: 2020-01-10T13:10:00+01:00
---
According to Wikipedia, "OAuth2 is an open standard for access delegation, used as a way for internet users to grant websites or applications access to their information on other websites but without giving them the passwords".  This is a nice simple definition, but since there are a number of actors involved and a number of possible levels of delegation, it's quite easy to get lost in the details.

To make things slightly more concrete, imagine an application (it could be a web app, a mobile app, or whatever - it's intentionally vague for now) that wants to do something with your list of Google contacts.  

In this situation, the roles from the [OAuth2 RFC](https://tools.ietf.org/html/rfc6749#section-1.1) would be as follows:
  * _resource owner_: you, the user of the "application"
  * _resource server_: a Google server somewhere that's capable of accepting access tokens and returning a list of Google contacts
  * _client_: the "application", which is going to want a list your google contacts
  * _authorisation server_: a Google server somewhere that's capable of issuing access tokens in response to authorised requests to do so.  This could be the same server as the resource server, or it could be different.

The OAuth2 protocol hinges on the idea of an _authorization grant_.  This represents what authorization the user (the _resource owner_) has given has given to the application (the _client_).  It comes in four basic types, although extensibility is built-in:
  * _authorization code_
  * _implicit_
  * _resource owner password credentials_
  * _client credentials_

In this post, I'll go through the use case of each of these _authorization grants_ types in turn.

## Authorization Code

A standard flow for the authorization code flow is:
  * _Resource owner_ visits _client_.  For example, you visit `https://contactlistapp.business`, which wants to know your Google contact list.  In this case, `https://contactlistapp.business` has a bunch of code running server-side, which is opaque to you.
  * That page (_client_) redirects the you (_resource owner_) to some Google server (_authorization server_).  It says that it is using the `authorization_code` flow, and passes along
    - A list of _scopes_ it wants.  This is going to be something encoding "I want to be able to read all of the contacts for this user"
    - A _redirect URL_, which is where the _authorization server_ should send the users browser back to once it has done its stuff
    - Additionally, it may pass along a _client ID_ and a _client secret_.  The _authorization server_ might require these in order to verify that the _client_ is a registered application.  Possession of the _client secret_ is deemed as proof that the request is indeed the _client_ with that _client id_, and not some other entity pretending to be it
  * The _authorization server_ will then perform some check that the user wants to grant these scopes to the application.  Conveniently, you have your browser pointed at the application server at this point, so for example it might ask you to enter your Google password
  * If the previous stage is successful, you the _authorization server_ redirects the _resource owner_ to the _redirect URL_.  It passes along the _authorization code_, 
  * The _client_, now in possession of an _authorization code_, can make another request to the _authorization server_ to exchange the _authorization code_ for an _access token_.  This is probably implemented as a callback on the handler for the _redirect URL_ that the _client_ initially provided
  * The _client_, now in possession of an _access token_, can make a request to the _resource server_ for the contact list


## Implicit

This skips the issuance of an authorization code.
  * _Resource owner_ visits _client_.  For example, you visit `https://contactlistapp.business`, which wants to know your Google contact list.  In this case, `https://contactlistapp.business` has all its logic in client-side Javascript, which is transparent to you
  * The _client_ redirects the _resource owner_ to the _authorization server_.  For example, you are redirected to some page on a Google domain, which asks if `contactlistapp` may access your contact list.  It says that it is using the `implicit` flow.  As before, it passes along the list of _scopes_ it desires, and the _redirect URL_ where you should end up when the authorization server has done its work

## Resource Owner Password Credentials

This is what it sounds like, and should obviously be used with extreme care.  If the _client_ is in posession of the the _resource owner_'s username and password, then hopefully it has been authorized to access the _resource owner_'s resources!  The protocol allows that the _authorization server_ can issue an access token in response to an authorization request of type `password_credentials`, if the _resource owner_'s password credentials are also provided

## Client Credentials

In this case, the _resource owner_ is not really involved at all.  This is purely a discussion between the _client_ and the _authorization server_.  Perhaps at some point in the past the _client_ and the _authorization server_ have agreed that there will be some resources on the _resource server_, possibly scoped to some _resource owner_ but possibly not, which the _authorization server_ will issue the _client_ access tokens for, so long as the _client_ can successfully authenticate itself.
